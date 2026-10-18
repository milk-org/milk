// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_proto.h
 * @brief UNIX domain socket wire protocol shared by milk-cli --server,
 *        libmilkbash.so, and milk-call.
 */

#ifndef MILK_PROTO_H
#define MILK_PROTO_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#define MILK_PROTO_MAGIC 0x4D494C4BU /* "MILK" */
#define MILK_PROTO_VERSION 1U

/* Request flags */
#define MILK_FLAG_ASYNC 0x01U
#define MILK_FLAG_BIND 0x02U   /* CALC: a line per statement, assignments too (@calc -v a,b) */
#define MILK_FLAG_PIXELS 0x04U /* CALC: as BIND, an image result as its pixels (@calc -a) */

/* Client -> Server message opcodes */
enum milk_msg_type
{
    MILK_MSG_PING     = 1,
    MILK_MSG_SHUTDOWN = 2,
    MILK_MSG_CALL     = 3,
    MILK_MSG_GET      = 4,
    MILK_MSG_SET      = 5,
    MILK_MSG_CALC     = 6,
    MILK_MSG_TEST     = 7,
    MILK_MSG_EXISTS   = 8,
    MILK_MSG_WAITFOR  = 9,
    MILK_MSG_GRAB     = 10,
    MILK_MSG_COMPLETE = 11,
    /* 12: LISTCMDS, removed (the install-time catalog lists the commands) */
    MILK_MSG_CANCEL = 13,
    MILK_MSG_WINCH  = 14,

    /* Server -> Client message opcodes */
    MILK_MSG_VALUE  = 101,
    MILK_MSG_SETVAR = 102,
    MILK_MSG_DEFCMD = 103,
    MILK_MSG_DONE   = 104
};

/**
 * @brief Frame header sent before every message payload.
 */
typedef struct
{
    uint32_t magic;
    uint16_t version;
    uint8_t  type;
    uint8_t  flags;
    uint32_t len;
} milk_frame_hdr_t;

/**
 * @brief Initialize sockaddr_un for either an abstract socket ("@name")
 *        or a filesystem socket path ("/tmp/...").
 *
 * @param sockpath Socket identifier string
 * @param addr     Output sockaddr_un struct
 * @param addrlen  Output socklen_t
 * @return 0 on success, -1 on error
 */
static inline int milk_sockaddr_init(const char         *sockpath,
                                     struct sockaddr_un *addr,
                                     socklen_t          *addrlen)
{
    if (sockpath == NULL || sockpath[0] == '\0')
    {
        return -1;
    }
    memset(addr, 0, sizeof(*addr));
    addr->sun_family = AF_UNIX;

    size_t slen = strlen(sockpath);
    if (sockpath[0] == '@')
    {
        if (slen >= sizeof(addr->sun_path))
        {
            return -1;
        }
        addr->sun_path[0] = '\0';
        memcpy(addr->sun_path + 1, sockpath + 1, slen - 1);
        *addrlen = (socklen_t) (offsetof(struct sockaddr_un, sun_path) + slen);
        return 0;
    }

    if (slen >= sizeof(addr->sun_path))
    {
        return -1;
    }
    memcpy(addr->sun_path, sockpath, slen + 1);
    *addrlen = (socklen_t) (offsetof(struct sockaddr_un, sun_path) + slen + 1);
    return 0;
}

/**
 * @brief Write all @len bytes to @fd, retrying on EINTR.
 */
static inline int milk_write_all(int fd, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *) buf;
    while (len > 0)
    {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return -1;
        }
        if (n == 0)
        {
            return -1;
        }
        p += n;
        len -= (size_t) n;
    }
    return 0;
}

/**
 * @brief Read exactly @len bytes from @fd, retrying on EINTR.
 */
static inline int milk_read_all(int fd, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *) buf;
    while (len > 0)
    {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return -1;
        }
        if (n == 0)
        {
            return -1;
        }
        p += n;
        len -= (size_t) n;
    }
    return 0;
}

/**
 * @brief Send a framed message without file descriptors.
 */
/**
 * milk_sendmsg_all() - send header + payload in one sendmsg (more only on a short write); @msg's
 * control data (fds) goes with the first chunk. 0 on success.
 */
static inline int milk_sendmsg_all(int fd, struct msghdr *msg, size_t total)
{
    size_t sent = 0;
    while (sent < total)
    {
        ssize_t n = sendmsg(fd, msg, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR)
        {
            continue;
        }
        if (n <= 0)
        {
            return -1;
        }
        sent += (size_t) n;
        msg->msg_control    = NULL;
        msg->msg_controllen = 0;
        while (msg->msg_iovlen > 0 && (size_t) n >= msg->msg_iov[0].iov_len)
        {
            n -= (ssize_t) msg->msg_iov[0].iov_len;
            msg->msg_iov++;
            msg->msg_iovlen--;
        }
        if (msg->msg_iovlen > 0)
        {
            msg->msg_iov[0].iov_base = (uint8_t *) msg->msg_iov[0].iov_base + n;
            msg->msg_iov[0].iov_len -= (size_t) n;
        }
    }
    return 0;
}

static inline int milk_proto_send(int         fd,
                                  uint8_t     type,
                                  uint8_t     flags,
                                  const void *payload,
                                  uint32_t    len)
{
    milk_frame_hdr_t hdr;
    hdr.magic   = MILK_PROTO_MAGIC;
    hdr.version = MILK_PROTO_VERSION;
    hdr.type    = type;
    hdr.flags   = flags;
    hdr.len     = len;

    struct iovec  iov[2] = { { &hdr, sizeof(hdr) }, { (void *) payload, len } };
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov    = iov;
    msg.msg_iovlen = (len > 0 && payload != NULL) ? 2 : 1;
    return milk_sendmsg_all(fd, &msg, sizeof(hdr) + ((msg.msg_iovlen == 2) ? len : 0));
}

/**
 * @brief Send a framed message along with 1..4 file descriptors via SCM_RIGHTS.
 */
static inline int milk_proto_send_with_fds(int         fd,
                                           uint8_t     type,
                                           uint8_t     flags,
                                           const void *payload,
                                           uint32_t    len,
                                           const int  *pass_fds,
                                           int         nfds)
{
    if (nfds < 1 || nfds > 4)
    {
        return -1;
    }

    milk_frame_hdr_t hdr;
    hdr.magic   = MILK_PROTO_MAGIC;
    hdr.version = MILK_PROTO_VERSION;
    hdr.type    = type;
    hdr.flags   = flags;
    hdr.len     = len;

    struct iovec iov[2] = { { &hdr, sizeof(hdr) }, { (void *) payload, len } };

    union
    {
        struct cmsghdr cm;
        char           control[CMSG_SPACE(sizeof(int) * 4)];
    } cmsg_un;

    memset(&cmsg_un, 0, sizeof(cmsg_un));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov        = iov;
    msg.msg_iovlen     = (len > 0 && payload != NULL) ? 2 : 1;
    msg.msg_control    = cmsg_un.control;
    msg.msg_controllen = CMSG_SPACE(sizeof(int) * (size_t) nfds);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level     = SOL_SOCKET;
    cmsg->cmsg_type      = SCM_RIGHTS;
    cmsg->cmsg_len       = CMSG_LEN(sizeof(int) * (size_t) nfds);
    memcpy(CMSG_DATA(cmsg), pass_fds, sizeof(int) * (size_t) nfds);

    return milk_sendmsg_all(fd, &msg, sizeof(hdr) + ((msg.msg_iovlen == 2) ? len : 0));
}

/**
 * @brief Receive a frame header (and optional SCM_RIGHTS fds) + payload.
 *
 * Caller must free(*payload_out) if *payload_out != NULL.
 */
/** milk_proto_close_fds() - close and reset received fds (error paths must never leak them). */
static inline void milk_proto_close_fds(int fds[4])
{
    for (int i = 0; fds != NULL && i < 4; i++)
    {
        if (fds[i] >= 0)
        {
            close(fds[i]);
            fds[i] = -1;
        }
    }
}

/**
 * milk_proto_recv() - read one frame; fds passed with it land in @recv_fds (or are closed).
 * Returns 0, -1 on I/O or framing error, or -2 if the peer speaks another protocol version.
 */
static inline int milk_proto_recv(int               fd,
                                  milk_frame_hdr_t *hdr_out,
                                  uint8_t         **payload_out,
                                  int               recv_fds[4])
{
    int got_fds[4] = { -1, -1, -1, -1 };
    if (recv_fds != NULL)
    {
        for (int i = 0; i < 4; i++)
        {
            recv_fds[i] = -1;
        }
    }
    *payload_out = NULL;

    struct iovec iov;
    iov.iov_base = hdr_out;
    iov.iov_len  = sizeof(*hdr_out);

    union
    {
        struct cmsghdr cm;
        char           control[CMSG_SPACE(sizeof(int) * 8)];
    } cmsg_un;

    memset(&cmsg_un, 0, sizeof(cmsg_un));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = cmsg_un.control;
    msg.msg_controllen = sizeof(cmsg_un.control);

    ssize_t n;
    do
    {
#ifdef MSG_CMSG_CLOEXEC
        n = recvmsg(fd, &msg, MSG_CMSG_CLOEXEC);
#else
        n = recvmsg(fd, &msg, 0);
#endif
    } while (n < 0 && errno == EINTR);

    if (n <= 0)
    {
        return -1;
    }

    /* take ownership of passed fds first, so every error path below can close them */
    for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL; cmsg = CMSG_NXTHDR(&msg, cmsg))
    {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
        {
            size_t nfds = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int);
            int    tmp[8];
            if (nfds > 8)
            {
                nfds = 8;
            }
            memcpy(tmp, CMSG_DATA(cmsg), nfds * sizeof(int));
            for (size_t i = 0; i < nfds; i++)
            {
                if (recv_fds != NULL && i < 4)
                {
                    got_fds[i] = tmp[i];
                }
                else if (tmp[i] >= 0)
                {
                    close(tmp[i]);
                }
            }
        }
    }

    if ((size_t) n < sizeof(*hdr_out) &&
        milk_read_all(fd, ((uint8_t *) hdr_out) + n, sizeof(*hdr_out) - (size_t) n) != 0)
    {
        milk_proto_close_fds(got_fds);
        return -1;
    }
    if (hdr_out->magic != MILK_PROTO_MAGIC)
    {
        milk_proto_close_fds(got_fds);
        return -1;
    }
    if (hdr_out->version != MILK_PROTO_VERSION)
    {
        milk_proto_close_fds(got_fds);
        return -2;
    }

    if (hdr_out->len > 0)
    {
        if (hdr_out->len > (64U * 1024U * 1024U))
        {
            milk_proto_close_fds(got_fds);
            return -1;
        }
        uint8_t *buf = (uint8_t *) malloc(hdr_out->len + 1);
        if (buf == NULL)
        {
            milk_proto_close_fds(got_fds);
            return -1;
        }
        if (milk_read_all(fd, buf, hdr_out->len) != 0)
        {
            free(buf);
            milk_proto_close_fds(got_fds);
            return -1;
        }
        buf[hdr_out->len] = '\0';
        *payload_out      = buf;
    }
    if (recv_fds != NULL)
    {
        memcpy(recv_fds, got_fds, sizeof(got_fds));
    }
    return 0;
}

#endif /* MILK_PROTO_H */
